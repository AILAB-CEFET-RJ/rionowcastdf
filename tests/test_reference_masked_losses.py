import copy
import unittest
import torch
from adapter.reference_masked_losses import regression_loss, masked_squared_error, MaskedResidualEDMLoss
from adapter.reference_masked_flow import train_step, validation_pass

class Regression(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.conv=torch.nn.Conv2d(2,1,3,padding=1)
    def forward(self,zero,x): return self.conv(x)

class Denoiser(torch.nn.Module):
    def __init__(self,channels=2):
        super().__init__(); self.conv=torch.nn.Conv2d(channels+1,1,3,padding=1); self.latent=None
    def forward(self,latent,condition,sigma):
        self.latent=latent.detach().clone()
        return self.conv(torch.cat((latent,condition),1))

class MaskTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        m=torch.ones(2,1,4,4,dtype=torch.bool); m[:,:,0,:]=False
        self.b=dict(input=torch.randn(2,2,4,4),target=torch.rand(2,1,4,4),mask=m)

    def test_manual_reduction_zero_gradient(self):
        p=torch.tensor([1.,2.,3.,4.]).reshape(1,1,2,2).requires_grad_()
        m=torch.tensor([True,False,True,False]).reshape_as(p)
        r=masked_squared_error(p,torch.zeros_like(p),m)
        self.assertEqual(r.loss.item(),5); r.loss.backward()
        torch.testing.assert_close(p.grad.flatten(),torch.tensor([1.,0.,3.,0.]))

    def test_all_valid_mse(self):
        self.b['mask'].fill_(True); net=Regression()
        expected=(net(torch.zeros_like(self.b['target']),self.b['input'])-self.b['target']).square().mean()
        torch.testing.assert_close(regression_loss(net,self.b).loss,expected)

    def test_regression_invariance(self):
        net=Regression(); first=regression_loss(net,self.b).loss; first.backward()
        grads=[p.grad.clone() for p in net.parameters()]
        changed=copy.deepcopy(self.b); changed['target'][~changed['mask']]=float('nan')
        net.zero_grad(); second=regression_loss(net,changed).loss; second.backward()
        torch.testing.assert_close(first,second)
        for p,g in zip(net.parameters(),grads): torch.testing.assert_close(p.grad,g)

    def test_diffusion_invariance_loss_input_gradients(self):
        for conditioning in (False,True):
            reg=Regression().eval().requires_grad_(False); den=Denoiser(3 if conditioning else 2)
            loss=MaskedResidualEDMLoss(hr_mean_conditioning=conditioning)
            sigma=torch.full((2,1,1,1),.5); noise=torch.randn_like(self.b['target'])
            first=loss(den,reg,self.b,sigma=sigma,noise=noise).loss; latent=den.latent.clone()
            first.backward(); grads=[p.grad.clone() for p in den.parameters()]
            changed=copy.deepcopy(self.b); changed['target'][~changed['mask']]=float('inf')
            den.zero_grad(); second=loss(den,reg,changed,sigma=sigma,noise=noise).loss; second.backward()
            torch.testing.assert_close(first,second); torch.testing.assert_close(latent,den.latent)
            torch.testing.assert_close(latent[~self.b['mask']],(.5*noise)[~self.b['mask']])
            for p,g in zip(den.parameters(),grads): torch.testing.assert_close(p.grad,g)
            self.assertTrue(all(p.grad is None for p in reg.parameters()))

    def test_valid_changes_matter(self):
        net=Regression(); first=regression_loss(net,self.b).loss
        self.b['target'][self.b['mask']]+=10
        self.assertNotEqual(first.item(),regression_loss(net,self.b).loss.item())

    def test_empty_batch_no_adam_update(self):
        net=Regression(); opt=torch.optim.AdamW(net.parameters(),lr=.01)
        train_step(net,opt,regression_loss,self.b)
        old=[p.clone() for p in net.parameters()]; steps=[v['step'].clone() for v in opt.state.values()]
        self.b['mask'].fill_(False); self.b['target'].fill_(float('nan'))
        self.assertFalse(train_step(net,opt,regression_loss,self.b)['updated'])
        for p,v in zip(net.parameters(),old): torch.testing.assert_close(p,v)
        for state,v in zip(opt.state.values(),steps): torch.testing.assert_close(state['step'],v)

    def test_empty_diffusion_no_forward(self):
        self.b['mask'].fill_(False); reg=Regression().eval().requires_grad_(False); den=Denoiser()
        self.assertIsNone(MaskedResidualEDMLoss()(den,reg,self.b).loss); self.assertIsNone(den.latent)

    def test_mixed_empty_counts(self):
        self.b['mask'][0]=False; r=regression_loss(Regression(),self.b)
        self.assertEqual(r.empty_samples,1); self.assertEqual(r.valid_pixels,12)

    def test_validation_global_weighting(self):
        net=Regression(); b=copy.deepcopy(self.b); b['mask'].fill_(False); b['mask'][:,:,1,1]=True
        a1,a2=regression_loss(net,self.b),regression_loss(net,b)
        expected=(a1.numerator+a2.numerator)/(a1.valid_pixels+a2.valid_pixels)
        out=validation_pass(net,regression_loss,[self.b,b]); self.assertTrue(net.training)
        self.assertAlmostEqual(out['loss'],expected.item(),places=6)

    def test_validation_empty_and_restore(self):
        net=Regression(); self.b['mask'].fill_(False)
        with self.assertRaises(ValueError): validation_pass(net,regression_loss,[self.b])
        self.assertTrue(net.training)

    def test_nonfinite_rejection(self):
        self.b['target'][0,0,1,1]=float('nan')
        with self.assertRaises(ValueError): regression_loss(Regression(),self.b)
        y=torch.ones(1,1,2,2)
        with self.assertRaises(ValueError): masked_squared_error(y*float('nan'),y,y.bool())

    def test_regression_freeze_and_sigma_checks(self):
        loss=MaskedResidualEDMLoss(); reg=Regression(); den=Denoiser()
        with self.assertRaises(ValueError): loss(den,reg,self.b)
        reg.eval().requires_grad_(False)
        with self.assertRaises(ValueError): loss(den,reg,self.b,sigma=torch.zeros(2,1,1,1))

    def test_rng_reproducibility(self):
        reg=Regression().eval().requires_grad_(False); den=Denoiser(); loss=MaskedResidualEDMLoss()
        a=loss(den,reg,self.b,generator=torch.Generator().manual_seed(42)).loss
        b=loss(den,reg,self.b,generator=torch.Generator().manual_seed(42)).loss
        torch.testing.assert_close(a,b)

    def test_all_valid_edm_formula(self):
        self.b['mask'].fill_(True); reg=Regression().eval().requires_grad_(False); den=Denoiser()
        sigma=torch.full((2,1,1,1),.5); noise=torch.randn_like(self.b['target'])
        residual=self.b['target']-reg(torch.zeros_like(self.b['target']),self.b['input'])
        expected=(8*(den(residual+.5*noise,self.b['input'],sigma)-residual).square()).mean()
        got=MaskedResidualEDMLoss()(den,reg,self.b,sigma=sigma,noise=noise).loss
        torch.testing.assert_close(got,expected)

if __name__=='__main__': unittest.main()
